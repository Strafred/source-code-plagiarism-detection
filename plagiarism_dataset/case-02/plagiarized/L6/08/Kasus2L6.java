import java.util.Scanner;
public class Kasus2L6 {
    public static void main(String[] args) {
        System.out.print("Enter the radius and length of a cylinder: ");
        Scanner in = new Scanner(System.in);
        float input = in.nextFloat();
        double luas = hitungLuas(input);
        input = in.nextFloat();
        System.out.println("The area is " + luas);
        System.out.println("The volume of the cylinder is " + hitungVolume(luas, input));
    }
    private static double hitungLuas(float radius){
        return radius * radius * 3.14159;
    }
    private static double hitungVolume(double luas, float length){
        return luas * length;
    }
}
