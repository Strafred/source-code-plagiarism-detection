import java.util.Scanner;
public class Main {
    private static double radius,length,areas,volumes;
    public static void main(String[] args) {
        System.out.print("Enter the radius and length of a cylinder: ");
        Scanner input = new Scanner(System.in);
        radius = input.nextDouble();
        length = input.nextDouble();
        areas = radius * radius * 3.14159;
        volumes = areas * length;
        cetak();
    }
    public static void cetak() {
        System.out.println("The area is " + areas);
        System.out.println("The volume of the cylinder is " + volumes);
    }
}
