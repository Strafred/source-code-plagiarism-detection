import java.util.Scanner;
public class Main {
    private static double weights,feets,inches,heights,bmi;
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter weight in pounds: ");
        weights = input.nextDouble();
        System.out.print("Enter feet: ");
        feets = input.nextDouble();
        System.out.print("Enter inches: ");
        inches = input.nextDouble();
        heights = feets * 12 + inches;
        bmi = weights * 0.45359237
        / ((heights * 0.0254) * (heights * 0.0254));
        cetak();
    }
    public static void cetak(){
        System.out.println("BMI is " + bmi); 
        if (bmi >= 30){
            System.out.println("Obese");
        } else if (bmi >= 25) {
            System.out.println("Overweight");
        }else if (bmi >= 18.5) {
            System.out.println("Normal");
        } else {
            System.out.println("Underweight");
        }
    }
}
